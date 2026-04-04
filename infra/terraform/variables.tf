variable "aws_region" {
  type        = string
  description = "AWS region for resources"
  default     = "us-east-1"
}

variable "name_prefix" {
  type        = string
  description = "Prefix for resource names (lowercase, no spaces)"
  default     = "embryo-admin"
}

variable "environment" {
  type        = string
  description = "Environment tag, e.g. dev, staging, prod"
  default     = "dev"
}
