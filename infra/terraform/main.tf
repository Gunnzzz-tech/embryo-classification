# Minimal example: private S3 bucket for Vite `dist/` artifacts.
# Pair with GitHub Actions OIDC + IAM role to run `aws s3 sync dist/ s3://...`
# and optionally invalidate CloudFront (add distribution in a follow-up module).

data "aws_caller_identity" "current" {}

locals {
  bucket_name = "${var.name_prefix}-frontend-${data.aws_caller_identity.current.account_id}-${var.environment}"
}

resource "aws_s3_bucket" "frontend" {
  bucket = local.bucket_name

  tags = {
    Name        = local.bucket_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}
