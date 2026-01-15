#!/usr/bin/env bash

function finetune {
    objective="$1"
    data="$2"
    checkpoint="facebook/esm2_t6_8M_UR50D"
    embed="$(basename "${checkpoint%.*}")"
    for action in default optim; do
        for size in full lora; do
            echo "Running finetuning step - ${objective} ${embed} ${action} ${size}"
            full=$(if [[ $size == "full" ]]; then echo "True"; else echo "False"; fi)
            outpath="results/${objective}_${embed}_${size}_${action}"
            python scripts/finetune.py \
                                -obj $objective -${action} -f $full -i $data \
                                -o $outpath -cp $checkpoint -nt1 3 -nt2 3       
        done
    done
}

# ---- dispatcher ----
COMMAND="$1"
shift  # remove command name from args

case "$COMMAND" in
    finetune)
        finetune "$@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: $0 finetune <arg1> <arg2>"
        exit 1
        ;;
esac

# Run the function with arguments in command line / terminal

# Supervised learning task
# sh scripts/finetune.sh finetune sv data/fluorescence.csv

# Masked language modeling task
# sh scripts/finetune.sh finetune mlm data/fluorescence_homologs.fasta