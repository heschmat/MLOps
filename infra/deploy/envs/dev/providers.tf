terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    bucket       = "heart-api-state-bucket"
    key          = "deploy/dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Environment = terraform.workspace
      ProjectName = var.project_name
    }
  }
}


locals {
  prefix = "${var.project_name}-${terraform.workspace}"
}

data "aws_region" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    # values = ["al2023-ami-*-x86_64"]
    # Full AL2023 AMIs do not contain -minimal- in this position:
    values = ["al2023-ami-2023.*-x86_64"]
  }

}