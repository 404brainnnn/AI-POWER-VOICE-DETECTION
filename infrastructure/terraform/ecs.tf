data "aws_availability_zones" "available" { state = "available" }

resource "aws_vpc" "main" { cidr_block = "10.42.0.0/16"; enable_dns_hostnames = true; enable_dns_support = true }
resource "aws_internet_gateway" "main" { vpc_id = aws_vpc.main.id }
resource "aws_subnet" "public" { count = 2; vpc_id = aws_vpc.main.id; cidr_block = "10.42.${count.index}.0/24"; availability_zone = data.aws_availability_zones.available.names[count.index]; map_public_ip_on_launch = true }
resource "aws_route_table" "public" { vpc_id = aws_vpc.main.id; route { cidr_block = "0.0.0.0/0"; gateway_id = aws_internet_gateway.main.id } }
resource "aws_route_table_association" "public" { count = 2; subnet_id = aws_subnet.public[count.index].id; route_table_id = aws_route_table.public.id }

resource "aws_security_group" "alb" { name = "${var.project_name}-alb"; vpc_id = aws_vpc.main.id; ingress { from_port = 80; to_port = 80; protocol = "tcp"; cidr_blocks = ["0.0.0.0/0"] }; egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] } }
resource "aws_security_group" "ecs" { name = "${var.project_name}-ecs"; vpc_id = aws_vpc.main.id; ingress { from_port = 8000; to_port = 8000; protocol = "tcp"; security_groups = [aws_security_group.alb.id] }; egress { from_port = 0; to_port = 0; protocol = "-1"; cidr_blocks = ["0.0.0.0/0"] } }

resource "aws_lb" "api" { name = substr("${var.project_name}-alb", 0, 32); load_balancer_type = "application"; subnets = aws_subnet.public[*].id; security_groups = [aws_security_group.alb.id] }
resource "aws_lb_target_group" "api" { name = substr("${var.project_name}-tg", 0, 32); port = 8000; protocol = "HTTP"; target_type = "ip"; vpc_id = aws_vpc.main.id; health_check { path = "/health"; matcher = "200" } }
resource "aws_lb_listener" "api" { load_balancer_arn = aws_lb.api.arn; port = 80; protocol = "HTTP"; default_action { type = "forward"; target_group_arn = aws_lb_target_group.api.arn } }

data "aws_iam_policy_document" "ecs_tasks_assume" { statement { actions = ["sts:AssumeRole"]; principals { type = "Service"; identifiers = ["ecs-tasks.amazonaws.com"] } } }
resource "aws_iam_role" "ecs_execution" { name = "${var.project_name}-execution"; assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json }
resource "aws_iam_role_policy_attachment" "ecs_execution" { role = aws_iam_role.ecs_execution.name; policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy" }
resource "aws_ecs_cluster" "main" { name = var.project_name }
resource "aws_cloudwatch_log_group" "ecs" { name = "/ecs/${var.project_name}/service"; retention_in_days = 30 }
resource "aws_ecs_task_definition" "api" { family = var.project_name; requires_compatibilities = ["FARGATE"]; network_mode = "awsvpc"; cpu = "512"; memory = "1024"; execution_role_arn = aws_iam_role.ecs_execution.arn; container_definitions = jsonencode([{ name = "api", image = "${aws_ecr_repository.api.repository_url}:latest", essential = true, portMappings = [{ containerPort = 8000, hostPort = 8000, protocol = "tcp" }], environment = [{ name = "APP_ENV", value = "production" }], logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.ecs.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "api" } } }]) }
resource "aws_ecs_service" "api" { name = var.project_name; cluster = aws_ecs_cluster.main.id; task_definition = aws_ecs_task_definition.api.arn; desired_count = 1; launch_type = "FARGATE"; network_configuration { subnets = aws_subnet.public[*].id; security_groups = [aws_security_group.ecs.id]; assign_public_ip = true }; load_balancer { target_group_arn = aws_lb_target_group.api.arn; container_name = "api"; container_port = 8000 }; depends_on = [aws_lb_listener.api] }

resource "aws_s3_bucket" "artifacts" { bucket_prefix = "${var.project_name}-artifacts-"; force_destroy = false }
resource "aws_s3_bucket_public_access_block" "artifacts" { bucket = aws_s3_bucket.artifacts.id; block_public_acls = true; block_public_policy = true; ignore_public_acls = true; restrict_public_buckets = true }
