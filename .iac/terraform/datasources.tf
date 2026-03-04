
data "azurerm_resource_group" "cocktails_resource_group" {
  name = "rg-${var.sub}-${var.region}-${var.environment}-${var.domain}-${var.sequence}"
}

data "azurerm_resource_group" "global_shared_resource_group" {
  name = "rg-${var.sub}-${var.region}-${var.global_environment}-shared-${var.sequence}"
}

data "azurerm_key_vault" "cocktails_keyvault" {
  name                = "kv-${var.sub}-${var.region}-${var.environment}-${var.shortdomain}-${var.short_sequence}"
  resource_group_name = data.azurerm_resource_group.cocktails_resource_group.name
}

data "azurerm_key_vault" "global_keyvault" {
  name                = "kv-${var.sub}-${var.region}-${var.global_environment}-shared-${var.short_sequence}"
  resource_group_name = data.azurerm_resource_group.global_shared_resource_group.name
}

data "azurerm_key_vault_secret" "otel_collector_onprem_key" {
  name         = "otel-collector-api-key-onprem"
  key_vault_id = data.azurerm_key_vault.global_keyvault.id
}