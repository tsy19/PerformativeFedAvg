import mlflow


def get_run_id(runs, pattern):
    run_id = runs.loc[runs['tags.mlflow.runName'].str.match(pattern), 'run_id'].item()
    return run_id


def add_tag(runs, tag):
    # import ipdb; ipdb.set_trace()
    for i in range(len(runs)):
        run_id = runs['run_id'].iloc[i]
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.set_tag('nickname', tag)


# st = pd.to_datetime(runs['start_time'])
# runs[(st > '2023-01-22 20:00') & (st < '2023-01-23 2:00')].shape