terraform {
  required_version = ">= 1.6.0"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.0" } }
}

variable "aws_region" { type = string, default = "us-east-1" }
variable "project_name" { type = string, default = "voice-clone-detector" }
provider "aws" { region = var.aws_region }

resource "aws_ecr_repository" "api" {
  name = var.project_name
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_cloudwatch_log_group" "api" { name = "/ecs/${var.project_name}" retention_in_days = 30 }
output "ecr_repository_url" { value = aws_ecr_repository.api.repository_url }
output "log_group_name" { value = aws_cloudwatch_log_group.api.name }
