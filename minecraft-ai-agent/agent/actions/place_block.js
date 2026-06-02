const Vec3 = require('vec3')
const { findInventoryItem, gotoNear, withTimeout } = require('./utils')

module.exports = async function placeBlock (bot, params) {
  const item = findInventoryItem(bot, params.block)
  if (!item) throw new Error(`No ${params.block} in inventory`)

  const target = new Vec3(Number(params.x), Number(params.y), Number(params.z))
  const reference = bot.blockAt(target.offset(0, -1, 0))
  if (!reference || reference.name === 'air') {
    throw new Error('Cannot place block without a solid block below target')
  }

  await gotoNear(bot, target, 4, 15000)
  await bot.equip(item, 'hand')
  await withTimeout(bot.placeBlock(reference, new Vec3(0, 1, 0)), 10000, 'place_block')
  return { ok: true, message: `Placed ${params.block}` }
}
