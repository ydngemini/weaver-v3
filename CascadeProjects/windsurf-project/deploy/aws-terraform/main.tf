locals {
  tags = {
    Project   = var.project_name
    Component = "weaver-brain"
    ManagedBy = "terraform"
  }
}

resource "aws_key_pair" "this" {
  key_name   = "${var.project_name}-key"
  public_key = file(pathexpand(var.public_key_path))
  tags       = local.tags
}

# Public edge firewall. On AWS the Security Group IS the firewall — Ubuntu AMIs ship no
# default iptables REJECT, so there is no on-box firewall step (unlike Oracle Cloud).
# ONLY 22 (locked) + 80/443 (Caddy) are open. Weaver's internal ports stay localhost-only:
#   9999 Nexus Bus · 8899 LoRA (binds 0.0.0.0!) · 8090 experts · 8000 backend · 9996/9997 dashboards.
resource "aws_security_group" "this" {
  name        = "${var.project_name}-edge-sg"
  description = "Weaver public edge: SSH (locked to operator) + HTTP/HTTPS for Caddy TLS only."
  vpc_id      = data.aws_vpc.default.id
  tags        = local.tags

  ingress {
    description = "SSH - locked to operator network(s)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.ssh_ingress_cidrs
  }

  ingress {
    description = "HTTP - Caddy ACME challenge and 80 to 443 redirect"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS - Caddy public TLS (Oracle UI and /ws)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound (apt, model pull, provider APIs, ACME)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "this" {
  ami                    = data.aws_ami.ubuntu_arm64.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.this.key_name
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.this.id]

  root_block_device {
    volume_type = "gp3"
    volume_size = var.root_volume_gb
    encrypted   = true
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required" # IMDSv2 only
  }

  tags = merge(local.tags, { Name = "${var.project_name}-brain" })
}

# Stable public IP so <EIP>.sslip.io + the Let's Encrypt cert survive reboots
# (fixes Oracle's ephemeral-IP problem).
resource "aws_eip" "this" {
  domain = "vpc"
  tags   = merge(local.tags, { Name = "${var.project_name}-eip" })
}

resource "aws_eip_association" "this" {
  instance_id   = aws_instance.this.id
  allocation_id = aws_eip.this.id
}
