variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "weaver"
}

variable "vm_size" {
  description = "Azure VM size (E4s_v5 = 4 vCPU / 32 GB memory-optimized)"
  type        = string
  default     = "Standard_E4s_v5"
}

variable "root_volume_gb" {
  description = "OS disk size in GB"
  type        = number
  default     = 128
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key (~/.ssh/id_ed25519.pub)"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_ingress_cidrs_v4" {
  description = "IPv4 CIDRs allowed to reach SSH (port 22). Prefer YOUR_IP/32"
  type        = string
}

variable "ssh_ingress_cidrs_v6" {
  description = "IPv6 CIDRs allowed to reach SSH (port 22). Prefer YOUR_IP/128"
  type        = string
  default     = "*"
}

variable "project_name" {
  description = "Name/tag prefix for resources"
  type        = string
  default     = "weaver"
}

variable "admin_username" {
  description = "VM admin username"
  type        = string
  default     = "ubuntu"
}

variable "dns_label" {
  description = "DNS label prefix for the public IP (optional)"
  type        = string
  default     = ""
}
