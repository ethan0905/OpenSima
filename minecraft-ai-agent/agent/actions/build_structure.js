const Vec3 = require('vec3')
const { findInventoryItem, gotoNear, withTimeout } = require('./utils')

function shelterOffsets () {
  const offsets = []
  for (let x = -1; x <= 1; x++) {
    for (let z = -1; z <= 1; z++) {
      if (Math.abs(x) === 1 || Math.abs(z) === 1) {
        offsets.push(new Vec3(x, 0, z))
        offsets.push(new Vec3(x, 1, z))
      }
      offsets.push(new Vec3(x, 2, z))
    }
  }
  return offsets
}

async function placeAt (bot, base, offset) {
  const target = base.plus(offset)
  const existing = bot.blockAt(target)
  if (!existing || existing.name !== 'air') return false

  const referenceOffsets = [
    new Vec3(0, -1, 0),
    new Vec3(1, 0, 0),
    new Vec3(-1, 0, 0),
    new Vec3(0, 0, 1),
    new Vec3(0, 0, -1)
  ]

  for (const refOffset of referenceOffsets) {
    const referencePos = target.plus(refOffset)
    const reference = bot.blockAt(referencePos)
    if (reference && reference.name !== 'air') {
      const face = target.minus(referencePos)
      await withTimeout(bot.placeBlock(reference, face), 10000, 'build_structure.place')
      return true
    }
  }

  return false
}

module.exports = async function buildStructure (bot, params) {
  if (params.structure !== 'basic_shelter') {
    throw new Error('Only basic_shelter is implemented for MVP')
  }

  let item = findInventoryItem(bot, params.material)
  if (!item) throw new Error(`No ${params.material} in inventory`)

  const base = bot.entity.position.floored()
  const offsets = shelterOffsets()
  const available = bot.inventory.items()
    .filter(invItem => invItem.name === params.material)
    .reduce((sum, invItem) => sum + invItem.count, 0)

  await gotoNear(bot, base, 2, 15000)
  let placed = 0
  const maxPlacements = Math.min(available, offsets.length)

  for (const offset of offsets.slice(0, maxPlacements)) {
    item = findInventoryItem(bot, params.material)
    if (!item) break
    await bot.equip(item, 'hand')
    const didPlace = await placeAt(bot, base, offset)
    if (didPlace) placed += 1
  }

  return {
    ok: placed >= offsets.length,
    partial: placed < offsets.length,
    placed,
    needed: offsets.length,
    message: placed >= offsets.length
      ? 'Built basic shelter'
      : `Built partial shelter with ${placed} of ${offsets.length} blocks`
  }
}
