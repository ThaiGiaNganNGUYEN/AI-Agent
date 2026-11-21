module "network" {
  source             = "./modules/network"
  name               = var.name
  cidr               = var.vpc_cidr
  availability_zones = var.availability_zones
}

module "registry" {
  source = "./modules/registry"
  name   = var.name
}

module "cluster" {
  source = "./modules/cluster"
  name   = var.name
}
