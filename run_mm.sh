
function run_parallel_jobs {
    local concurrent_max=$1
    local callback=$2
    local cmds=("${@:3}")
    local jobs=( )

    while [[ "${#cmds[@]}" -gt 0 ]] || [[ "${#jobs[@]}" -gt 0 ]]; do
        while [[ "${#jobs[@]}" -lt $concurrent_max ]] && [[ "${#cmds[@]}" -gt 0 ]]; do
            local cmd="${cmds[0]}"
            cmds=("${cmds[@]:1}")

            bash -c "$cmd" &
            jobs+=($!)
        done

        local job="${jobs[0]}"
        jobs=("${jobs[@]:1}")

        local state="$(ps -p $job -o state= 2>/dev/null)"

        if [[ "$state" == "D" ]] || [[ "$state" == "Z" ]]; then
            $callback $job
        else
            wait $job
            $callback $job $?
        fi
    done
}

function job_done {
    if [[ $# -lt 2 ]]; then
        echo "PID $1 died unexpectedly"
    else
        echo "PID $1 exited $2"
    fi
}

cmds=()



tr_v=(0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.)
compliances_v=(1.)
cATE_v=(0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.)

for tr in "${tr_v[@]}"; do
  for compliances in "${compliances_v[@]}"; do
    for cATE in "${cATE_v[@]}"; do
      cmds+=("python -m umm.compute_metrics_mm --cATE $cATE --tr $tr --comp $compliances --output mm_result --n_split 51")
      done
  done
done



cpus=40
run_parallel_jobs $cpus "job_done" "${cmds[@]}"
