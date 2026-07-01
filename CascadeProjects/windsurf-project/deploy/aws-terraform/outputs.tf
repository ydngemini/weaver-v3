output "public_ip" {
  description = "Elastic IP of the Weaver box."
  value       = aws_eip.this.public_ip
}

output "hostname" {
  description = "Zero-signup public hostname — sslip.io resolves it to the EIP and Let's Encrypt issues for it. Use this as ORACLE_HOST in setup_oracle_extras.sh."
  value       = "${aws_eip.this.public_ip}.sslip.io"
}

output "ssh_command" {
  description = "SSH into the box."
  value       = "ssh ubuntu@${aws_eip.this.public_ip}"
}

output "ami_id" {
  description = "Resolved Ubuntu 24.04 arm64 AMI."
  value       = data.aws_ami.ubuntu_arm64.id
}
