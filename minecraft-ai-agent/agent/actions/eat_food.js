const { findInventoryItem, withTimeout } = require('./utils')

const KNOWN_FOODS = [
  'cooked_beef',
  'cooked_porkchop',
  'cooked_chicken',
  'bread',
  'apple',
  'carrot',
  'potato',
  'baked_potato',
  'beef',
  'porkchop',
  'chicken'
]

module.exports = async function eatFood (bot, params) {
  if (bot.food >= 18) {
    return { ok: true, skipped: true, message: 'Food level is already high' }
  }

  const item = params.food
    ? findInventoryItem(bot, params.food)
    : KNOWN_FOODS.map(name => findInventoryItem(bot, name)).find(Boolean)

  if (!item) throw new Error('No edible food found in inventory')

  await bot.equip(item, 'hand')
  await withTimeout(bot.consume(), 10000, 'eat_food')
  return { ok: true, message: `Ate ${item.name}` }
}
