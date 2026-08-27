output "public_ip" {
  description = "Public IP of the Weaver VM"
  value       = azurerm_public_ip.this.ip_address
}

output "fqdn" {
  description = "FQDN if dns_label was set"
  value       = azurerm_public_ip.this.fqdn
}

output "ssh_command" {
  description = "SSH into the box"
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.this.ip_address}"
}

output "resource_group" {
  description = "Resource group name"
  value       = azurerm_resource_group.this.name
}

output "location" {
  description = "Azure region"
  value       = azurerm_resource_group.this.location
}

output "vm_id" {
  description = "VM resource ID"
  value       = azurerm_linux_virtual_machine.this.id
}

output "public_urls" {
  description = "Canonical Weaver URLs (set DNS A records to public_ip)"
  value = {
    embodiment = "https://weaverv3.com"
    headless   = "https://headless.weaverv3.com"
    dashboard  = "https://dash.weaverv3.com"
    status     = "https://status.weaverv3.com"
    brain      = "https://weaverv3.com/brain"
    tts        = "https://weaverv3.com/tts"
    codebase   = "https://weaverv3.com/codebase"
  }
}
