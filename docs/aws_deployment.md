# AWS deployment

Terraform now includes a parameterized GPU ECS EC2 path using an ECS GPU-optimized AMI, private subnets, NAT, ALB idle timeout, optional ACM TLS, ECR, CloudWatch, S3 model-artifact access, service autoscaling, and unhealthy-host alarms. Supply `allowed_security_group_ids`, `acm_certificate_arn`, and model artifact values for a private VoIP deployment. Secrets Manager/SSM signed-token or mTLS wiring remains environment-specific and must be added before production. A SageMaker variant is intentionally represented as a future module flag rather than created automatically: ECS EC2 gives control over persistent GPU capacity and WebSockets; SageMaker gives managed deployment/autoscaling but adds endpoint constraints and cost/operational coupling.

No AWS resource was created. Run `terraform -chdir=infrastructure/terraform fmt -check`, `terraform init`, and `terraform validate`, then require explicit approval for `terraform plan/apply`.

Build/publish is manual through `.github/workflows/ecr.yml` and requires an OIDC role in `AWS_DEPLOY_ROLE_ARN`. Install `requirements-deployment.txt` when using `scripts/resolve_model_artifact.py`. Pin the image tag to the commit SHA and set `model_checkpoint_uri` to a versioned S3 object; promotion must update Terraform/ECS explicitly rather than silently following a registry `latest` model.
