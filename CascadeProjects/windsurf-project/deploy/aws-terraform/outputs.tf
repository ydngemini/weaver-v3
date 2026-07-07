output "public_ip" {
  description = "Elastic IP of the Weaver box."
  value       = aws_eip.this.public_ip
}

output "hostname" {
  description = "Zero-signup public hostname — sslip.io resolves it to the EIP and Let's Encrypt issues for it. Use this as ORACLE_HOST in setup_oracle_extras.sh."
  value       = "${aws_eip.this.public_ip}.sslip.io"
}

output "public_urls" {
  description = "Canonical Weaver production URLs."
  value = {
    embodiment         = "https://weaverv3.com"
    headless           = "https://headless.weaverv3.com"
    dashboard          = "https://dash.weaverv3.com"
    status             = "https://status.weaverv3.com"
    brain              = "https://weaverv3.com/brain"
    realtime_voice     = "wss://weaverv3.com/brain/realtime/voice"
    text_to_speech     = "https://weaverv3.com/tts"
    readonly_codebase  = "https://weaverv3.com/codebase"
    bootstrap_hostname = "https://${aws_eip.this.public_ip}.sslip.io"
  }
}

output "ssh_command" {
  description = "SSH into the box."
  value       = "ssh ubuntu@${aws_eip.this.public_ip}"
}

output "ami_id" {
  description = "Resolved Ubuntu 24.04 arm64 AMI."
  value       = data.aws_ami.ubuntu_arm64.id
}
