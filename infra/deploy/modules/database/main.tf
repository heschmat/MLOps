resource "aws_db_subnet_group" "this" {
  name       = "${var.name}-subnet-group"
  subnet_ids = var.private_subnets
}

resource "aws_security_group" "this" {
  name   = "${var.name}-rds-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_groups
    # cidr_blocks = [ var.vpc_cidr ]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "this" {
  identifier = "${var.name}-pg"

  engine         = "postgres"
  engine_version = "18.1"

  instance_class = var.instance_class

  allocated_storage = 20

  db_name  = var.db_name
  username = var.username
  password = var.password

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.this.id]

  storage_encrypted       = true
  backup_retention_period = 7
  skip_final_snapshot     = true
  deletion_protection     = false
  multi_az = false
  apply_immediately = true
}