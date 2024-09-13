## Initialize sweep

This can be done on any machine with the code. Simply cd to the tree_neiborghs_match folder and run:

```
wandb sweep config.yaml
```

Take note of the sweep id (the one with your username and project name)

## Create conda enviornment for sweep

1. cd into tree_neightborghs_match folder on modi.
2. run the command:

   ```
   modi-new-job --generate-job-scripts --generate-container-wrap init_env.sh
   ```

3. When the output file indicates so, the envioronment is finished setting up.

## initialize a bunch of agents on modi:

1. Insert your wandb api key in the run_agents.sh file
2. run the command:

   ```
   sbatch -N NUM_NODES --ntasks NUM_AGENTS run_agents.sh.container_wrap
   ```

Where NUM_NODES is the amount of nodes to run on (there are 8) and NUM_AGENTS is the number of agents to run. This has been tested with 10 agents spread roughly over 2 nodes and works reasonably well.

## More

You may check the status of running jobs with squeue, and you may just a run down with scancel <job-id>
