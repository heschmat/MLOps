variable "name" {
  type = string
}

variable "image" {
  type = string
}

variable "cpu" {
  type    = string
  default = "512"
}

variable "memory" {
  type    = string
  default = "1024"
}

variable "container_port" {
  type    = number
  default = 5000
}

variable "env_vars" {
  type    = map(string)
  default = {}
}

variable "subnets" {
  type = list(string)
}

variable "vpc_id" {
  type = string
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "cluster_id" {
  type = string
}

variable "execution_role_arn" {
  type = string
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "command" {
  type    = list(string)
  default = null
}
