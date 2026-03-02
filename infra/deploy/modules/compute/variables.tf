variable "name" {}
variable "vpc_id" {}
variable "vpc_cidr" {}
variable "subnet_id" {}

variable "ami_id" {}
variable "instance_type" {
  default = "t3.micro"
}