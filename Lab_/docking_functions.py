from pathlib import Path

import numpy as np
import pandas as pd
import py3Dmol

from IPython.display import HTML
from vina import Vina


HOST_ORDER = ["BCD", "HPBCD", "SBEBCD"]
FORM_ORDER = ["B", "C"]
GUEST_NAME = "5,7-dimethoxyflavone"
PREPARED_INPUT_VERSION = "prepared_folder_loader_v03"
VINA_BOX_SIZE_A = np.array([20.0, 20.0, 20.0], dtype=float)

ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.974,
    "S": 32.060,
    "F": 18.998,
    "Cl": 35.450,
    "Br": 79.904,
    "I": 126.900,
    "Na": 22.990,
}

POSE_OVERLAY_COLORS = ["#1f77b4", "#4c78a8", "#72b7b2", "#54a24b", "#eeca3b", "#f58518", "#e45756", "#b279a2"]
FORM_PANEL_COLORS = {"B": "#1f77b4", "C": "#f58518"}

PREP_DIR: Path | None = None
RESULTS_DIR: Path | None = None

prepared_files = None
prepared_files_version = None
host_boxes = None
box_table = None
published_pose_outputs = None
published_pose_table = None


def configure_paths(prep_dir: Path, results_dir: Path) -> None:
    global PREP_DIR, RESULTS_DIR
    global prepared_files, prepared_files_version
    global host_boxes, box_table, published_pose_outputs, published_pose_table

    PREP_DIR = Path(prep_dir)
    RESULTS_DIR = Path(results_dir)
    prepared_files = None
    prepared_files_version = None
    host_boxes = None
    box_table = None
    published_pose_outputs = None
    published_pose_table = None


def _require_configured_paths() -> tuple[Path, Path]:
    if PREP_DIR is None or RESULTS_DIR is None:
        raise RuntimeError("Call configure_paths(prep_dir, results_dir) before using docking functions.")
    return PREP_DIR, RESULTS_DIR


def atomic_masses(elements) -> np.ndarray:
    return np.array([ATOMIC_MASSES.get(el, 12.0) for el in elements], dtype=float)


def center_of_mass(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    masses = atomic_masses(df["element"])
    return np.average(coords, axis=0, weights=masses)


def build_box_from_host(host_df: pd.DataFrame):
    center = center_of_mass(host_df)
    return center.astype(float), VINA_BOX_SIZE_A.copy()


def autodock_type_to_element(ad_type: str, atom_name: str) -> str:
    ad_type = ad_type.strip()
    atom_name = atom_name.strip()
    if ad_type.startswith("Cl") or atom_name.startswith("Cl"):
        return "Cl"
    if ad_type.startswith("Br") or atom_name.startswith("Br"):
        return "Br"
    if ad_type.startswith("Na") or atom_name.startswith("Na"):
        return "Na"
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if letters:
        if len(letters) >= 2 and letters[:2].capitalize() in {"Cl", "Br", "Na"}:
            return letters[:2].capitalize()
        return letters[0].upper()
    if ad_type:
        return ad_type[0].upper()
    return "C"


def load_pdb_atoms(pdb_path: Path) -> pd.DataFrame:
    rows = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            element = line[76:78].strip() or "".join(ch for ch in atom_name if ch.isalpha())[:1].upper() or "C"
            rows.append((element, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return pd.DataFrame(rows, columns=["element", "x", "y", "z"])


def parse_pdbqt_models(pdbqt_path: Path) -> list[pd.DataFrame]:
    models = []
    current_rows = []
    saw_model = False
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith("MODEL"):
            saw_model = True
            current_rows = []
            continue
        if line.startswith("ENDMDL"):
            if current_rows:
                models.append(pd.DataFrame(current_rows, columns=["element", "x", "y", "z", "atom_name", "ad_type"]))
            current_rows = []
            continue
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ad_type = line.split()[-1]
            element = autodock_type_to_element(ad_type, atom_name)
            current_rows.append((element, x, y, z, atom_name, ad_type))
    if current_rows:
        models.append(pd.DataFrame(current_rows, columns=["element", "x", "y", "z", "atom_name", "ad_type"]))
    if saw_model:
        return models
    return models[:1]


def parse_vina_results(pdbqt_path: Path) -> pd.DataFrame:
    rows = []
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            parts = line.split()
            rows.append({
                "affinity_kcal_mol": float(parts[3]),
                "rmsd_lb_A": float(parts[4]),
                "rmsd_ub_A": float(parts[5]),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table.insert(0, "mode", np.arange(1, len(table) + 1))
    return table


def host_axis(host_df: pd.DataFrame) -> np.ndarray:
    coords = host_df[["x", "y", "z"]].to_numpy(dtype=float)
    centered = coords - center_of_mass(host_df)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmin(eigvals)]
    return axis / np.linalg.norm(axis)


def group_center(df: pd.DataFrame) -> np.ndarray:
    if len(df) == 0:
        raise ValueError("Cannot compute a center for an empty atom group")
    return center_of_mass(df)


def guest_orientation_descriptor(host_df: pd.DataFrame, guest_df: pd.DataFrame) -> dict:
    axis = host_axis(host_df)
    host_com = center_of_mass(host_df)
    guest_com = center_of_mass(guest_df)
    oxygen_df = guest_df[guest_df["element"] == "O"]
    carbon_df = guest_df[guest_df["element"] == "C"]
    oxygen_com = group_center(oxygen_df) if len(oxygen_df) else guest_com
    carbon_com = group_center(carbon_df) if len(carbon_df) else guest_com
    return {
        "guest_com_axis_A": float(np.dot(guest_com - host_com, axis)),
        "oxygen_minus_carbon_axis_A": float(np.dot(oxygen_com - carbon_com, axis)),
    }


def pose_dataframe_to_xyz(pose_df: pd.DataFrame, comment: str) -> str:
    rows = pose_df[["element", "x", "y", "z"]].itertuples(index=False)
    body = "\n".join(
        f"{row.element:<2} {row.x: .6f} {row.y: .6f} {row.z: .6f}"
        for row in rows
    )
    return f"{len(pose_df)}\n{comment}\n{body}\n"


def py3dmol_view_to_html(view) -> str:
    if hasattr(view, "_make_html"):
        return view._make_html()
    if hasattr(view, "_repr_html_"):
        return view._repr_html_()
    raise RuntimeError("Could not convert the py3Dmol view to HTML")


def compose_py3dmol_views(views, gap_px: int = 12) -> HTML:
    html_parts = [
        "<div style='display:flex; flex-wrap:nowrap; align-items:flex-start; gap:{}px;'>".format(gap_px)
    ]
    for view in views:
        html_parts.append("<div style='flex:1 1 0; min-width:0;'>")
        html_parts.append(py3dmol_view_to_html(view))
        html_parts.append("</div>")
    html_parts.append("</div>")
    return HTML("".join(html_parts))


def compose_py3dmol_row(left_view, right_view, gap_px: int = 12) -> HTML:
    return compose_py3dmol_views([left_view, right_view], gap_px=gap_px)


def prepared_path(filename: str) -> Path:
    prep_dir, _ = _require_configured_paths()
    path = prep_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required prepared file not found: {path}. This notebook only loads files from Docking_Vina/prepared."
        )
    return path


def ensure_prepared_files():
    global prepared_files, prepared_files_version
    if (
        prepared_files is None
        or prepared_files_version != PREPARED_INPUT_VERSION
    ):
        prepared = {
            "57-DMF": {
                "name": GUEST_NAME,
                "pdb": prepared_path("57-DMF_from_smiles.pdb"),
                "pdbqt": prepared_path("57-DMF_flex.pdbqt"),
            }
        }
        for host in HOST_ORDER:
            prepared[host] = {
                "pdb": prepared_path(f"{host}.pdb"),
                "pdbqt": prepared_path(f"{host}_rigid.pdbqt"),
            }
        for host in HOST_ORDER:
            for form in FORM_ORDER:
                pose_name = f"57-DMF-{host}_{form}-form"
                prepared[pose_name] = {
                    "complex_pdb": prepared_path(f"{pose_name}.pdb"),
                    "host_pdb": prepared_path(f"{pose_name}_host.pdb"),
                    "guest_pdb": prepared_path(f"{pose_name}_guest.pdb"),
                }
        prepared_files = prepared
        prepared_files_version = PREPARED_INPUT_VERSION
    return prepared_files


def ensure_host_boxes():
    global host_boxes, box_table
    if host_boxes is None or box_table is None:
        prepared = ensure_prepared_files()
        box_rows = []
        host_box_map = {}
        for host in HOST_ORDER:
            host_df = load_pdb_atoms(prepared[host]["pdb"])
            center, size = build_box_from_host(host_df)
            host_box_map[host] = (center, size)
            box_rows.append({
                "host": host,
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
            })
        host_boxes = host_box_map
        box_table = pd.DataFrame(box_rows).round(3)
    return host_boxes, box_table


def ensure_published_pose_outputs():
    global published_pose_outputs, published_pose_table
    if published_pose_outputs is None or published_pose_table is None:
        prepared = ensure_prepared_files()
        published_pose_rows = []
        pose_outputs = {}
        for host in HOST_ORDER:
            for form in FORM_ORDER:
                pose_name = f"57-DMF-{host}_{form}-form"
                pose_outputs[(host, form)] = {
                    "host_pdb": prepared[pose_name]["host_pdb"],
                    "guest_pdb": prepared[pose_name]["guest_pdb"],
                }
                published_pose_rows.append({
                    "host": host,
                    "form": form,
                    "published_host_pdb": str(prepared[pose_name]["host_pdb"]),
                    "published_guest_pdb": str(prepared[pose_name]["guest_pdb"]),
                })
        published_pose_outputs = pose_outputs
        published_pose_table = pd.DataFrame(published_pose_rows)
        published_pose_table["form"] = pd.Categorical(published_pose_table["form"], FORM_ORDER, ordered=True)
        published_pose_table = published_pose_table.sort_values(["host", "form"]).reset_index(drop=True)
    return published_pose_outputs, published_pose_table


def vina_box_pdb_block(center, size) -> str:
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    half = size / 2.0
    corners = [
        center + np.array([-half[0], -half[1], -half[2]]),
        center + np.array([-half[0], -half[1], half[2]]),
        center + np.array([-half[0], half[1], -half[2]]),
        center + np.array([-half[0], half[1], half[2]]),
        center + np.array([half[0], -half[1], -half[2]]),
        center + np.array([half[0], -half[1], half[2]]),
        center + np.array([half[0], half[1], -half[2]]),
        center + np.array([half[0], half[1], half[2]]),
    ]
    edges = [
        (1, 2), (1, 3), (1, 5),
        (2, 4), (2, 6),
        (3, 4), (3, 7),
        (5, 6), (5, 7),
        (4, 8), (6, 8), (7, 8),
    ]
    header = ["VINA_BOX", "Codex", ""]
    counts = f"{8:>3}{12:>3}  0  0  0  0            999 V2000"
    atom_lines = [
        f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0"
        for coord in corners
    ]
    bond_lines = [
        f"{a:>3}{b:>3}{1:>3}  0  0  0  0"
        for a, b in edges
    ]
    return "\n".join(header + [counts] + atom_lines + bond_lines + ["M  END", "$$$$"]) + "\n"


def translate_pdb_block(pdb_path: Path, shift) -> str:
    shift = np.asarray(shift, dtype=float)
    shifted_lines = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + shift[0]
            y = float(line[38:46]) + shift[1]
            z = float(line[46:54]) + shift[2]
            shifted_lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
        else:
            shifted_lines.append(line)
    return "\n".join(shifted_lines) + "\n"


def centered_ligand_pdb_block(guest_pdb: Path, target_center) -> str:
    guest_df = load_pdb_atoms(guest_pdb)
    shift = np.asarray(target_center, dtype=float) - center_of_mass(guest_df)
    return translate_pdb_block(guest_pdb, shift)


def build_initial_docking_setup_panel(host: str, prepared_files: dict, host_boxes: dict, width: int = 760, height: int = 430):
    center, size = host_boxes[host]
    host_pdb = prepared_files[host]["pdb"]
    guest_pdb = prepared_files["57-DMF"]["pdb"]
    ligand_block = centered_ligand_pdb_block(guest_pdb, center)

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    view.addModel(vina_box_pdb_block(center, size), "mol")
    view.setStyle({"model": 0}, {"stick": {"radius": 0.10, "color": "#d62728"}})
    view.addModel(host_pdb.read_text(), "pdb")
    view.setStyle({"model": 1}, {"stick": {"radius": 0.16, "colorscheme": "grayCarbon"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.10, "color": "lightgray"}, {"model": 1})
    view.addModel(ligand_block, "pdb")
    view.setStyle({"model": 2}, {"stick": {"radius": 0.22, "colorscheme": "greenCarbon"}, "sphere": {"scale": 0.18, "colorscheme": "greenCarbon"}})
    view.addLabel(
        f"{host} pre-docking setup\nStandalone host + ligand before B/C classification\n20 x 20 x 20 A Vina box",
        {
            "fontSize": 14,
            "backgroundColor": "white",
            "fontColor": "#2c7c31",
            "showBackground": True,
            "inFront": True,
        },
    )
    view.zoomTo()
    view.zoom(0.72)
    return view


def show_initial_docking_setup(host: str, prepared_files: dict, host_boxes: dict, width: int = 760, height: int = 430):
    return build_initial_docking_setup_panel(host, prepared_files, host_boxes, width=width, height=height)


def classify_docked_modes_by_form(host: str, docked_pdbqt: Path, modes: pd.DataFrame) -> pd.DataFrame:
    prepared = ensure_prepared_files()
    published_outputs, _ = ensure_published_pose_outputs()
    host_df = load_pdb_atoms(prepared[host]["pdb"])
    template_vectors = {}
    for form in FORM_ORDER:
        guest_df = load_pdb_atoms(published_outputs[(host, form)]["guest_pdb"])
        desc = guest_orientation_descriptor(host_df, guest_df)
        template_vectors[form] = np.array([
            desc["guest_com_axis_A"],
            desc["oxygen_minus_carbon_axis_A"],
        ], dtype=float)
    scale = np.maximum(np.abs(template_vectors["B"] - template_vectors["C"]), 1.0)
    pose_models = parse_pdbqt_models(docked_pdbqt)
    if len(pose_models) != len(modes):
        raise RuntimeError(
            f"Mismatch between parsed docked pose models ({len(pose_models)}) and Vina score rows ({len(modes)}) for {host}"
        )
    rows = []
    for mode_row, pose_df in zip(modes.to_dict("records"), pose_models):
        desc = guest_orientation_descriptor(host_df, pose_df)
        pose_vector = np.array([
            desc["guest_com_axis_A"],
            desc["oxygen_minus_carbon_axis_A"],
        ], dtype=float)
        distance_to_B = float(np.linalg.norm((pose_vector - template_vectors["B"]) / scale))
        distance_to_C = float(np.linalg.norm((pose_vector - template_vectors["C"]) / scale))
        assigned_form = "B" if distance_to_B <= distance_to_C else "C"
        rows.append({
            "host": host,
            **mode_row,
            "assigned_form": assigned_form,
            "guest_com_axis_A": desc["guest_com_axis_A"],
            "oxygen_minus_carbon_axis_A": desc["oxygen_minus_carbon_axis_A"],
            "distance_to_B": distance_to_B,
            "distance_to_C": distance_to_C,
        })
    return pd.DataFrame(rows)


def summarize_form_distribution(redocking_outputs: dict) -> tuple[dict, pd.DataFrame]:
    pose_tables = {}
    summary_rows = []
    for host in HOST_ORDER:
        pose_table = classify_docked_modes_by_form(host, redocking_outputs[host]["docked_pdbqt"], redocking_outputs[host]["modes"])
        pose_tables[host] = pose_table
        total = len(pose_table)
        for form in FORM_ORDER:
            subset = pose_table[pose_table["assigned_form"] == form]
            percent_dc = 100.0 * len(subset) / total if total else np.nan
            summary_rows.append({
                "host": host,
                "form": form,
                "%DC": percent_dc,
                "mean_vina_interaction_energy_kcal_mol": float(subset["affinity_kcal_mol"].mean()) if len(subset) else np.nan,
                "best_vina_interaction_energy_kcal_mol": float(subset["affinity_kcal_mol"].min()) if len(subset) else np.nan,
                "n_poses": int(len(subset)),
            })
    summary = pd.DataFrame(summary_rows)
    summary["form"] = pd.Categorical(summary["form"], FORM_ORDER, ordered=True)
    summary["host"] = pd.Categorical(summary["host"], HOST_ORDER, ordered=True)
    summary = summary.sort_values(["host", "form"]).reset_index(drop=True)
    return pose_tables, summary


def dock_host(host: str, box_center, box_size, exhaustiveness: int = 24, n_poses: int = 20):
    _, results_dir = _require_configured_paths()
    prepared = ensure_prepared_files()
    ligand_path = prepared["57-DMF"]["pdbqt"]
    v = Vina(sf_name="vina")
    v.set_receptor(str(prepared[host]["pdbqt"]))
    v.set_ligand_from_file(str(ligand_path))
    v.compute_vina_maps(center=box_center.tolist(), box_size=box_size.tolist())
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    docked_path = results_dir / f"{host}_redocked_out.pdbqt"
    v.write_poses(str(docked_path), n_poses=n_poses, overwrite=True)
    modes = parse_vina_results(docked_path)
    if modes.empty:
        raise RuntimeError(f"Vina returned no docked poses for {host}")
    best_affinity = float(modes.iloc[0]["affinity_kcal_mol"])
    return {
        "host": host,
        "best_docked_affinity_kcal_mol": best_affinity,
        "best_mode_rmsd_lb_A": float(modes.iloc[0]["rmsd_lb_A"]),
        "best_mode_rmsd_ub_A": float(modes.iloc[0]["rmsd_ub_A"]),
        "n_reported_modes": len(modes),
        "docked_pdbqt": docked_path,
        "modes": modes,
    }


def build_host_pose_interaction_panel(host: str, form: str, prepared_files: dict, pose_tables_by_host: dict, redocking_outputs: dict, width: int = 480, height: int = 430):
    pose_models = parse_pdbqt_models(redocking_outputs[host]["docked_pdbqt"])
    host_pdb = prepared_files[host]["pdb"]
    view = py3Dmol.view(width=width, height=height)
    view.addModel(host_pdb.read_text(), "pdb")
    view.setStyle({"model": 0}, {"stick": {"radius": 0.16, "colorscheme": "grayCarbon"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.10, "color": "lightgray"}, {"model": 0})
    subset = pose_tables_by_host[host][pose_tables_by_host[host]["assigned_form"] == form].sort_values("affinity_kcal_mol").reset_index(drop=True)
    if subset.empty:
        view.addLabel(
            f"{host} {form}-form\nNo assigned poses",
            {"fontSize": 14, "backgroundColor": "white", "fontColor": FORM_PANEL_COLORS[form], "showBackground": True},
        )
        view.zoomTo()
        return view
    for pose_idx, row in enumerate(subset.itertuples(index=False), start=1):
        model_df = pose_models[int(row.mode) - 1]
        color = POSE_OVERLAY_COLORS[(pose_idx - 1) % len(POSE_OVERLAY_COLORS)]
        view.addModel(pose_dataframe_to_xyz(model_df, f"{host} {form} mode {row.mode}"), "xyz")
        view.setStyle({"model": pose_idx}, {"stick": {"radius": 0.14, "color": color}})
    best_energy = subset["affinity_kcal_mol"].min()
    view.addLabel(
        f"{host} {form}-form\n{subset.shape[0]} poses\nBest {best_energy:.2f} kcal/mol",
        {"fontSize": 14, "backgroundColor": "white", "fontColor": FORM_PANEL_COLORS[form], "showBackground": True},
    )
    view.zoomTo()
    return view


def show_host_pose_interaction_grid(host: str, prepared_files: dict, pose_tables_by_host: dict, redocking_outputs: dict, width: int = 980, height: int = 430):
    panel_width = max(int((width - 12) / 2), 420)
    left_view = build_host_pose_interaction_panel(host, "B", prepared_files, pose_tables_by_host, redocking_outputs, width=panel_width, height=height)
    right_view = build_host_pose_interaction_panel(host, "C", prepared_files, pose_tables_by_host, redocking_outputs, width=panel_width, height=height)
    return compose_py3dmol_row(left_view, right_view)


def build_best_pose_panel(host: str, form: str, prepared_files: dict, pose_tables_by_host: dict, redocking_outputs: dict, width: int = 480, height: int = 430):
    pose_models = parse_pdbqt_models(redocking_outputs[host]["docked_pdbqt"])
    host_pdb = prepared_files[host]["pdb"]
    view = py3Dmol.view(width=width, height=height)
    view.addModel(host_pdb.read_text(), "pdb")
    view.setStyle({"model": 0}, {"stick": {"radius": 0.16, "colorscheme": "grayCarbon"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.10, "color": "lightgray"}, {"model": 0})
    subset = pose_tables_by_host[host][pose_tables_by_host[host]["assigned_form"] == form].sort_values("affinity_kcal_mol").reset_index(drop=True)
    if subset.empty:
        view.addLabel(
            f"{host} {form}-form\nNo assigned poses",
            {"fontSize": 14, "backgroundColor": "white", "fontColor": FORM_PANEL_COLORS[form], "showBackground": True},
        )
        view.zoomTo()
        return view
    best_row = subset.iloc[0]
    best_model = pose_models[int(best_row["mode"]) - 1]
    view.addModel(pose_dataframe_to_xyz(best_model, f"{host} best {form} mode {int(best_row['mode'])}"), "xyz")
    view.setStyle({"model": 1}, {"stick": {"radius": 0.24, "color": FORM_PANEL_COLORS[form]}, "sphere": {"scale": 0.20, "color": FORM_PANEL_COLORS[form]}})
    view.addLabel(
        f"{host} best {form}-form\nMode {int(best_row['mode'])}\n{best_row['affinity_kcal_mol']:.2f} kcal/mol",
        {"fontSize": 14, "backgroundColor": "white", "fontColor": FORM_PANEL_COLORS[form], "showBackground": True},
    )
    view.zoomTo()
    return view


def show_best_pose_grid(host: str, prepared_files: dict, pose_tables_by_host: dict, redocking_outputs: dict, width: int = 980, height: int = 430):
    panel_width = max(int((width - 12) / 2), 420)
    left_view = build_best_pose_panel(host, "B", prepared_files, pose_tables_by_host, redocking_outputs, width=panel_width, height=height)
    right_view = build_best_pose_panel(host, "C", prepared_files, pose_tables_by_host, redocking_outputs, width=panel_width, height=height)
    return compose_py3dmol_row(left_view, right_view)
