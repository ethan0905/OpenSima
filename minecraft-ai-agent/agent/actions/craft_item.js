const { withTimeout } = require('./utils')

module.exports = async function craftItem (bot, params) {
  const itemName = params.item
  const count = Number(params.count ?? 1)
  const item = bot.registry.itemsByName[itemName]
  if (!item) throw new Error(`Unknown item: ${itemName}`)

  let craftingTable = null
  const nearbyTable = bot.findBlock({
    matching: block => block && block.name === 'crafting_table',
    maxDistance: 6
  })

  const inventoryRecipes = bot.recipesFor(item.id, null, 1, null)
  if (inventoryRecipes.length === 0 && nearbyTable) {
    craftingTable = nearbyTable
  }

  const recipes = bot.recipesFor(item.id, null, 1, craftingTable)
  if (!recipes.length) {
    throw new Error(`No available recipe for ${itemName}. Missing materials or crafting table.`)
  }

  await withTimeout(bot.craft(recipes[0], count, craftingTable), 30000, 'craft_item')
  return { ok: true, message: `Crafted ${count} ${itemName}` }
}
