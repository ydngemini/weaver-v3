# Weaver avatar asset bucket — public-read static assets (GLBs, embodiment page, renders).
# The embodiment page is fully client-side, so serving it from S3's HTTPS endpoint gives
# a secure context (camera/mic allowed) on any device. Brain/Nexus calls stay localhost
# on the viewer's machine and degrade gracefully when absent.
data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "avatar" {
  bucket = "${var.project_name}-avatar-${data.aws_caller_identity.current.account_id}"
  tags   = local.tags
}

resource "aws_s3_bucket_public_access_block" "avatar" {
  bucket                  = aws_s3_bucket.avatar.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = false # policy below grants read-only GetObject
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "avatar_read" {
  bucket     = aws_s3_bucket.avatar.id
  depends_on = [aws_s3_bucket_public_access_block.avatar]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "PublicReadObjects"
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.avatar.arn}/*"
    }]
  })
}

resource "aws_s3_bucket_cors_configuration" "avatar" {
  bucket = aws_s3_bucket.avatar.id
  cors_rule {
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = ["*"]
    allowed_headers = ["*"]
    max_age_seconds = 86400
  }
}

output "avatar_bucket" {
  value = aws_s3_bucket.avatar.id
}

output "avatar_embodiment_url" {
  value = "https://${aws_s3_bucket.avatar.id}.s3.amazonaws.com/embodiment.html"
}
