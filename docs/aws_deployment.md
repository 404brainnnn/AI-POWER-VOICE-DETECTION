# AWS deployment

Terraform now defines a baseline VPC, public ALB/ECS Fargate service, ECR, CloudWatch, and private S3 artifact bucket. External Redis/MongoDB-compatible endpoints and Secrets Manager wiring remain deployment-specific and must be added before production. No AWS resource was created. Run `terraform -chdir=infrastructure/terraform fmt -check`, `terraform init`, and `terraform validate`, then require explicit approval for `terraform plan/apply`.
