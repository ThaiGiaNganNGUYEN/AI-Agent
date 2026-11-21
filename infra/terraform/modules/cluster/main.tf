variable "name" { type = string }

resource "aws_cloudwatch_log_group" "this" {
  name              = "/agentflow/${var.name}"
  retention_in_days = 30
}

resource "aws_ecs_cluster" "this" {
  name = "${var.name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

output "cluster_name" { value = aws_ecs_cluster.this.name }
output "log_group_name" { value = aws_cloudwatch_log_group.this.name }
