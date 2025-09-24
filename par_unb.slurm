    #!/bin/bash
    #SBATCH --array=1-30
    #SBATCH --partition=batch
    #SBATCH -J test
    #SBATCH -o displays/test.%A.%a.out
    #SBATCH -e displays/test.%A.%a.err
    #SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa
    #SBATCH --mail-type=ALL
    #SBATCH --time=5-00:00:00
    #SBATCH --mem=200G
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=40


#run the application:
    #source ~/miniconda3/bin/activate loc 
    export CONDA_PKGS_DIRS=/ibex/users/alvarem/conda_cache
    source /ibex/user/alvarem/miniconda3/bin/activate conda activate /home/alvarem/miniconda3/envs/loc
    Array=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
    ARGUMENT=${Array[$SLURM_ARRAY_TASK_ID-1]}
    python pfiles/Par_loc_GCSL3_ip_ii.py "$ARGUMENT"

