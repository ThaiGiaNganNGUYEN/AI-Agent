variable "aws_region" {
  description = "AWS region. Availability zones below must belong to this region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Prefix for AgentFlow resources."
  type        = string
  default     = "agentflow"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "availability_zones" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b"]
}
