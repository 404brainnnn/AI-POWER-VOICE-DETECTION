data "aws_ssm_parameter" "ecs_gpu_ami" { name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id" }

variable "deployment_mode" { type = string; default = "ec2_gpu"; validation { condition = contains(["ec2_gpu", "fargate"], var.deployment_mode); error_message = "deployment_mode must be ec2_gpu or fargate" } }
variable "gpu_instance_type" { type = string; default = "g4dn.xlarge" }
variable "gpu_instance_min_size" { type = number; default = 1 }
variable "gpu_instance_desired_size" { type = number; default = 1 }
variable "gpu_instance_max_size" { type = number; default = 4 }
variable "allowed_security_group_ids" { type = list(string); default = [] }
variable "allowed_cidr_blocks" { type = list(string); default = [] }
variable "acm_certificate_arn" { type = string; default = "" }
variable "model_artifact_bucket" { type = string; default = "" }
variable "model_checkpoint_uri" { type = string; default = "" }
variable "api_key_secret_arn" { type = string; default = "" }
locals { artifact_bucket = var.model_artifact_bucket != "" ? var.model_artifact_bucket : aws_s3_bucket.artifacts.bucket }

resource "aws_subnet" "private" { count = 2; vpc_id = aws_vpc.main.id; cidr_block = "10.42.${count.index + 10}.0/24"; availability_zone = data.aws_availability_zones.available.names[count.index] }
resource "aws_eip" "nat" { domain = "vpc" }
resource "aws_nat_gateway" "main" { allocation_id = aws_eip.nat.id; subnet_id = aws_subnet.public[0].id }
resource "aws_route_table" "private" { vpc_id = aws_vpc.main.id; route { cidr_block = "0.0.0.0/0"; nat_gateway_id = aws_nat_gateway.main.id } }
resource "aws_route_table_association" "private" { count = 2; subnet_id = aws_subnet.private[count.index].id; route_table_id = aws_route_table.private.id }

resource "aws_security_group" "gpu_ecs" { name = "${var.project_name}-gpu-ecs"; vpc_id = aws_vpc.main.id; ingress { from_port = 8000; to_port = 8000; protocol = "tcp"; security_groups = [aws_security_group.alb.id] }; egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] } }

resource "aws_iam_role" "ecs_instance" { name = "${var.project_name}-instance"; assume_role_policy = data.aws_iam_policy_document.ecs_instance_assume.json }
data "aws_iam_policy_document" "ecs_instance_assume" { statement { actions = ["sts:AssumeRole"]; principals { type = "Service"; identifiers = ["ec2.amazonaws.com"] } } }
resource "aws_iam_role_policy_attachment" "ecs_instance" { role = aws_iam_role.ecs_instance.name; policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role" }
resource "aws_iam_instance_profile" "ecs" { name = "${var.project_name}-instance"; role = aws_iam_role.ecs_instance.name }

resource "aws_launch_template" "gpu" { name_prefix = "${var.project_name}-gpu-"; image_id = data.aws_ssm_parameter.ecs_gpu_ami.value; instance_type = var.gpu_instance_type; iam_instance_profile { name = aws_iam_instance_profile.ecs.name }; vpc_security_group_ids = [aws_security_group.gpu_ecs.id]; user_data = base64encode("#!/bin/bash\necho ECS_CLUSTER=${aws_ecs_cluster.main.name} >> /etc/ecs/ecs.config\necho ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config\n"); tag_specifications { resource_type = "instance"; tags = { Name = var.project_name } } }
resource "aws_autoscaling_group" "gpu" { name = "${var.project_name}-gpu"; min_size = var.gpu_instance_min_size; desired_capacity = var.gpu_instance_desired_size; max_size = var.gpu_instance_max_size; vpc_zone_identifier = aws_subnet.private[*].id; launch_template { id = aws_launch_template.gpu.id; version = "$Latest" }; protect_from_scale_in = true }
resource "aws_ecs_capacity_provider" "gpu" { name = "${var.project_name}-gpu"; auto_scaling_group_provider { auto_scaling_group_arn = aws_autoscaling_group.gpu.arn; managed_termination_protection = "ENABLED"; managed_scaling { status = "ENABLED"; target_capacity = 75 } } }
resource "aws_ecs_cluster_capacity_providers" "gpu" { cluster_name = aws_ecs_cluster.main.name; capacity_providers = [aws_ecs_capacity_provider.gpu.name]; default_capacity_provider_strategy { capacity_provider = aws_ecs_capacity_provider.gpu.name; weight = 1 } }

resource "aws_lb_target_group" "api_gpu" { name = substr("${var.project_name}-gpu-tg", 0, 32); port = 8000; protocol = "HTTP"; target_type = "instance"; vpc_id = aws_vpc.main.id; health_check { path = "/healthz"; matcher = "200" } }
resource "aws_ecs_task_definition" "api_gpu" { family = "${var.project_name}-gpu"; requires_compatibilities = ["EC2"]; network_mode = "bridge"; cpu = "2048"; memory = "8192"; execution_role_arn = aws_iam_role.ecs_execution.arn; task_role_arn = aws_iam_role.ecs_task.arn; container_definitions = jsonencode([{ name = "api", image = "${aws_ecr_repository.api.repository_url}:latest", essential = true, resourceRequirements = [{ type = "GPU", value = "1" }], portMappings = [{ containerPort = 8000, hostPort = 0, protocol = "tcp" }], environment = [{ name = "APP_ENV", value = "production" }, { name = "MODEL_ARTIFACT_BUCKET", value = local.artifact_bucket }, { name = "MODEL_CHECKPOINT", value = var.model_checkpoint_uri }], secrets = var.api_key_secret_arn == "" ? [] : [{ name = "API_KEY", valueFrom = var.api_key_secret_arn }], logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.ecs.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "gpu-api" } } }]) }
resource "aws_ecs_service" "api_gpu" { name = "${var.project_name}-gpu"; cluster = aws_ecs_cluster.main.id; task_definition = aws_ecs_task_definition.api_gpu.arn; desired_count = 1; capacity_provider_strategy { capacity_provider = aws_ecs_capacity_provider.gpu.name; weight = 1 }; deployment_circuit_breaker { enable = true; rollback = true }; health_check_grace_period_seconds = 120; load_balancer { target_group_arn = aws_lb_target_group.api_gpu.arn; container_name = "api"; container_port = 8000 }; depends_on = [aws_ecs_cluster_capacity_providers.gpu] }

data "aws_iam_policy_document" "ecs_task_assume" { statement { actions = ["sts:AssumeRole"]; principals { type = "Service"; identifiers = ["ecs-tasks.amazonaws.com"] } } }
resource "aws_iam_role" "ecs_task" { name = "${var.project_name}-task"; assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json }
resource "aws_iam_role_policy" "ecs_task_s3" { name = "${var.project_name}-s3-read"; role = aws_iam_role.ecs_task.id; policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["s3:GetObject", "s3:ListBucket"], Resource = ["arn:aws:s3:::${local.artifact_bucket}", "arn:aws:s3:::${local.artifact_bucket}/*"] }] }) }

resource "aws_appautoscaling_target" "ecs" { max_capacity = 4; min_capacity = 1; resource_id = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api_gpu.name}"; scalable_dimension = "ecs:service:DesiredCount"; service_namespace = "ecs" }
resource "aws_appautoscaling_policy" "ecs_cpu" { name = "${var.project_name}-cpu" policy_type = "TargetTrackingScaling" resource_id = aws_appautoscaling_target.ecs.resource_id scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension service_namespace = aws_appautoscaling_target.ecs.service_namespace target_tracking_scaling_policy_configuration { target_value = 60; predefined_metric_specification { predefined_metric_type = "ECSServiceAverageCPUUtilization" }; scale_in_cooldown = 120; scale_out_cooldown = 60 } }
resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" { alarm_name = "${var.project_name}-unhealthy-hosts" namespace = "AWS/ApplicationELB" metric_name = "UnHealthyHostCount" dimensions = { TargetGroup = aws_lb_target_group.api_gpu.arn_suffix; LoadBalancer = aws_lb.api.arn_suffix } statistic = "Maximum" period = 60 evaluation_periods = 3 threshold = 1 comparison_operator = "GreaterThanOrEqualToThreshold" }
