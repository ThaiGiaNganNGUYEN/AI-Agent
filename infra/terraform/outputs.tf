output "vpc_id" {
  value = module.network.vpc_id
}

output "public_subnet_ids" {
  value = module.network.public_subnet_ids
}

output "ecr_repository_url" {
  value = module.registry.repository_url
}

output "ecs_cluster_name" {
  value = module.cluster.cluster_name
}

output "log_group_name" {
  value = module.cluster.log_group_name
}
