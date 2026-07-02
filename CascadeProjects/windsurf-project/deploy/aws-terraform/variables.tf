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

variable "ssh_ingress_cidrs" {
  description = "CIDRs allowed to reach SSH (port 22). Prefer YOUR_IP/32; a rotating-IP ISP may need your provider's /16(s). NEVER 0.0.0.0/0."
  type        = list(string)

  validation {
    condition     = length(var.ssh_ingress_cidrs) > 0 && alltrue([for c in var.ssh_ingress_cidrs : can(cidrhost(c, 0))])
    error_message = "ssh_ingress_cidrs must be a non-empty list of valid CIDRs, e.g. [\"203.0.113.4/32\"]."
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
