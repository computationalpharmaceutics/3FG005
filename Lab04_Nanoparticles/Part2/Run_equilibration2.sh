#!/bin/bash
set -euo pipefail

# step6.1
#gmx grompp -f step6.1_minimization.mdp -o step6.1_minimization.tpr -c step6.0_minimization.gro -r fullerenes_water.gro -p system.top -n index.ndx
#gmx mdrun -deffnm step6.1_minimization

cnt=2
cntmax=6 ;# 6

while [ "$cnt" -le "$cntmax" ]; do
    pcnt=$((cnt - 1))

    # first equilibration uses the minimization output
    if [ "$cnt" -eq 2 ]; then
        gmx grompp \
            -f "step6.${cnt}_equilibration.mdp" \
            -o "step6.${cnt}_equilibration.tpr" \
            -c "step6.${pcnt}_minimization.gro" \
            -r "F16_and_membrane.gro" \
            -p "system.top" \
            -n "index.ndx" -maxwarn 5
    else
        gmx grompp \
            -f "step6.${cnt}_equilibration.mdp" \
            -o "step6.${cnt}_equilibration.tpr" \
            -c "step6.${pcnt}_equilibration.gro" \
            -r "F16_and_membrane.gro" \
            -p "system.top" \
            -n "index.ndx" -maxwarn 5
    fi

    gmx mdrun -deffnm "step6.${cnt}_equilibration"
    cnt=$((cnt + 1))
done
