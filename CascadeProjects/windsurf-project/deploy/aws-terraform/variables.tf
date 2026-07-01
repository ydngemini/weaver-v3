variable "region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (Graviton/ARM). t4g.large = 2 vCPU / 8 GB — fits fully-local llama.cpp experts + Soul Voice + the brain."
  type        = string
  default     = "t4g.large"
}

variable "root_volume_gb" {
  description = "Encrypted gp3 root volume size (GB). Holds two GGUF models + two Python venvs + the Vite node build."
  type        = number
  default     = 50
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to reach SSH (port 22). Set to YOUR_IP/32 (curl ifconfig.me). NEVER 0.0.0.0/0."
  type        = string

  validation {
    condition     = can(cidrhost(var.ssh_ingress_cidr, 0))
    error_message = "ssh_ingress_cidr must be a valid CIDR, e.g. 203.0.113.4/32."
  }
}

variable "public_key_path" {
  description = "Path to the SSH public key installed on the box for the ubuntu user."
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "project_name" {
  description = "Name/tag prefix for all resources."
  type        = string
  default     = "weaver"
}
