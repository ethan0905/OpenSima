const { gotoNear, sleep } = require('./utils')

module.exports = async function attackEntity (bot, params) {
  const maxDistance = Number(params.max_distance ?? 16)
  const candidates = Object.values(bot.entities)
    .filter(entity => {
      if (!entity.position || entity === bot.entity) return false
      const name = entity.name || entity.mobType || entity.type
      return name === params.entity_type && bot.entity.position.distanceTo(entity.position) <= maxDistance
    })
    .sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position))

  if (!candidates.length) throw new Error(`No ${params.entity_type} within ${maxDistance} blocks`)

  const target = candidates[0]
  await gotoNear(bot, target.position, 3, 15000)
  bot.attack(target)
  await sleep(750)
  return { ok: true, message: `Attacked nearest ${params.entity_type}` }
}
