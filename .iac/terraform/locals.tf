locals {
  tags = {
    Product     = var.product
    Environment = var.environment
    Application = var.domain
    Owner       = var.owner
  }
}
