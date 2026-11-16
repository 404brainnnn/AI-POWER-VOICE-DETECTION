# Terraform

The checked-in Terraform is a safe starting point containing only an ECR repository and CloudWatch log group; it does not create networking, ECS, databases, or incur the broader production footprint. Run `terraform -chdir=infrastructure/terraform fmt -check` and `terraform -chdir=infrastructure/terraform validate` after installing Terraform. Never run apply without explicit approval, reviewed IAM/network rules, and configured secrets.
