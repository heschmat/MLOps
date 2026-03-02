output "compute_instance_id" {
  value = module.compute.instance_id
}

output "compute_sg_id" {
  value = module.compute.security_group_id
}

output "rds_endpoint" {
  value = module.database.endpoint
}