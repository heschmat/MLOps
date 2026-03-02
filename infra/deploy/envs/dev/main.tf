module "networking" {
  source = "../../modules/networking"

  name     = "${var.project_name}-vpc"
  vpc_cidr = "10.0.0.0/16"
  azs     = data.aws_availability_zones.available.names
}

module "compute" {
  source = "../../modules/compute"

  name      = "${var.project_name}-dev-test"
  vpc_id    = module.networking.vpc_id
  vpc_cidr  = module.networking.vpc_cidr
  subnet_id = module.networking.private_subnets[0]

  ami_id = data.aws_ami.amazon_linux.id
}

module "database" {
  source = "../../modules/database"

  name = "${var.project_name}-dev-db"

  vpc_id          = module.networking.vpc_id
  private_subnets = module.networking.private_subnets

  # vpc_cidr = module.networking.vpc_cidr
  allowed_security_groups = [
    module.compute.security_group_id,
    module.mlflow.security_group_id,
  ]

  db_name  = "heart_disease"
  username = "admino"
  password = var.db_password

  instance_class = "db.t3.micro" # db.t4g.micro
}

module "ecs_cluster" {
  source = "../../modules/ecs-cluster"
  name   = "${var.project_name}-ecs-cluster"
}


module "nginx" {
  source           = "../../modules/ecs-service"
  name             = "${var.project_name}-nginx"
  image            = "nginx:stable"
  cpu              = "256"
  memory           = "512"
  container_port   = 80
  subnets          = module.networking.public_subnets
  vpc_id           = module.networking.vpc_id
  desired_count    = 1
  cluster_id       = module.ecs_cluster.cluster_id
  aws_region       = "us-east-1"
  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  env_vars         = {}
}


module "mlflow" {
  source = "../../modules/ecs-service"

  name               = "${var.project_name}-mlflow"
  image              = "ghcr.io/mlflow/mlflow:v3.9.0" # official image
  cpu                = "1024"
  memory             = "2048"
  container_port     = 5000
  subnets            = module.networking.public_subnets
  vpc_id             = module.networking.vpc_id
  desired_count      = 1
  cluster_id         = module.ecs_cluster.cluster_id
  aws_region         = "us-east-1"

  command = [
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--allowed-hosts", "*" # For dev, allow all; for prod, specify your domain or IP
  ]

  execution_role_arn = aws_iam_role.ecs_execution_role.arn

  env_vars = {
    MLFLOW_BACKEND_STORE_URI = "postgresql://admino:${var.db_password}@${module.database.endpoint}:5432/mlflow"
    MLFLOW_ARTIFACT_ROOT     = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
    AWS_REGION               = "us-east-1"
  }
}

# misc

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-mlflow-artifacts"
}