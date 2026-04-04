output "frontend_bucket_name" {
  description = "Upload `npm run build` output here (e.g. aws s3 sync dist/ s3://BUCKET --delete)"
  value       = aws_s3_bucket.frontend.bucket
}

output "frontend_bucket_arn" {
  value = aws_s3_bucket.frontend.arn
}
