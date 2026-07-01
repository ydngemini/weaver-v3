# Single-box deploy — reuse the account's default VPC + subnets (no custom VPC needed;
# the box sits behind Caddy TLS, unlike the Neoh multi-tier zero-trust VPC).
data "aws_vpc" "default" {
  default = true
}

# Only consider AZs that actually offer the chosen instance type — e.g. us-east-1e
# has no Graviton, and a blind subnets[0] pick can land there and fail RunInstances.
data "aws_ec2_instance_type_offerings" "supported" {
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
  location_type = "availability-zone"
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.supported.locations
  }
}

# Latest Ubuntu 24.04 (Noble) arm64 server AMI, published by Canonical (owner 099720109477).
# The hvm-ssd* glob matches both the older hvm-ssd and newer hvm-ssd-gp3 image families.
data "aws_ami" "ubuntu_arm64" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd*/ubuntu-noble-24.04-arm64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["arm64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}
