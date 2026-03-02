variable "aws_region" {
  default = "us-east-1"
}

variable "project_name" {
  default = "heart-disease-api"
  type    = string
}

variable "db_password" {
  type      = string
  sensitive = true
  default = "**ChangeM3"
}
