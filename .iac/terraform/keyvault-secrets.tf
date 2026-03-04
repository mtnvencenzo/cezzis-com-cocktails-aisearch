resource "azurerm_key_vault_secret" "cocktails_aisearch_tei_embeddings_apikey" {
  name         = "cocktails-aisearch-tei-embeddings-apikey"
  value        = "n/a"
  key_vault_id = data.azurerm_key_vault.cocktails_keyvault.id

  lifecycle {
    ignore_changes = [value]
  }

  tags = local.tags
}

resource "azurerm_key_vault_secret" "cocktails_aisearch_tei_reranker_apikey" {
  name         = "cocktails-aisearch-tei-reranker-apikey"
  value        = "n/a"
  key_vault_id = data.azurerm_key_vault.cocktails_keyvault.id

  lifecycle {
    ignore_changes = [value]
  }

  tags = local.tags
}

resource "azurerm_key_vault_secret" "cocktails_aisearch_tei_splade_apikey" {
  name         = "cocktails-aisearch-tei-splade-apikey"
  value        = "n/a"
  key_vault_id = data.azurerm_key_vault.cocktails_keyvault.id

  lifecycle {
    ignore_changes = [value]
  }

  tags = local.tags
}
