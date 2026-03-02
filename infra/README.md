
```sh
docker compose run --rm --entrypoint sh tf


```


```sh
terraform state show module.compute.aws_security_group.this

terraform state show module.networking.module.vpc.aws_vpc.this[0]
```


```sh
aws ssm start-session --target <private-instance-id>

# DB
sudo dnf install postgresql17 -y

psql \
  -h $DB_ENDPOINT \
  -U admino \
  -d heart_disease \
  -p 5432

SELECT now();

## for mlflow
CREATE DATABASE mlflow;
CREATE USER mlflow_user WITH PASSWORD '#MLisFun';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO mlflow_user;


\c mlflow
```



## ecs

```sh
aws ecs list-clusters

aws ecs list-services --cluster $CLUSTER_NAME


aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME



# force new deployment 

aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment

## now you should get true for the new TASK:
aws ecs describe-tasks \
  --cluster $CLUSTER_NAME \
  --tasks $TASK_ID \
  --query "tasks[].enableExecuteCommand"

# get container name
aws ecs describe-tasks \
  --cluster $CLUSTER_NAME \
  --tasks $TASK_ID \
  --query 'tasks[0].containers[].name'

aws ecs execute-command \
  --cluster $CLUSTER_NAME \
  --task $TASK_ID \
  --region us-east-1 \
  --container $CONTAINER_NAME \
  --interactive \
  --command "/bin/sh"

```