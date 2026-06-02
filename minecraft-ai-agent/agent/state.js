const USEFUL_BLOCKS = new Set([
  'oak_log',
  'birch_log',
  'spruce_log',
  'stone',
  'dirt',
  'coal_ore',
  'iron_ore',
  'crafting_table',
  'furnace',
  'chest',
  'water',
  'lava'
])

function compactPosition (pos) {
  if (!pos) return null
  return {
    x: Math.round(pos.x * 100) / 100,
    y: Math.round(pos.y * 100) / 100,
    z: Math.round(pos.z * 100) / 100
  }
}

function inventorySummary (bot) {
  const counts = new Map()
  for (const item of bot.inventory.items()) {
    counts.set(item.name, (counts.get(item.name) || 0) + item.count)
  }
  return [...counts.entries()].map(([name, count]) => ({ name, count }))
}

function nearbyEntities (bot, radius = 24) {
  if (!bot.entity) return []

  return Object.values(bot.entities)
    .filter(entity => entity !== bot.entity && entity.position)
    .map(entity => ({
      name: entity.name || entity.username || entity.displayName || 'unknown',
      type: entity.type || 'unknown',
      distance: Math.round(bot.entity.position.distanceTo(entity.position) * 100) / 100,
      position: compactPosition(entity.position)
    }))
    .filter(entity => entity.distance <= radius)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 20)
}

function players (bot) {
  if (!bot.entity) return []

  return Object.values(bot.players)
    .filter(player => player.username && player.username !== bot.username)
    .map(player => ({
      username: player.username,
      distance: player.entity
        ? Math.round(bot.entity.position.distanceTo(player.entity.position) * 100) / 100
        : null,
      position: player.entity ? compactPosition(player.entity.position) : null
    }))
    .sort((a, b) => {
      if (a.distance === null) return 1
      if (b.distance === null) return -1
      return a.distance - b.distance
    })
}

function nearbyBlocks (bot, radius = 8) {
  if (!bot.entity) return []

  const blocks = bot.findBlocks({
    matching: block => block && USEFUL_BLOCKS.has(block.name),
    maxDistance: radius,
    count: 80
  })

  return blocks
    .map(pos => {
      const block = bot.blockAt(pos)
      return {
        name: block.name,
        distance: Math.round(bot.entity.position.distanceTo(pos) * 100) / 100,
        position: compactPosition(pos)
      }
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 40)
}

function getState (bot, isSpawned) {
  const position = bot.entity ? compactPosition(bot.entity.position) : { x: 0, y: 0, z: 0 }
  return {
    agent: {
      username: bot.username,
      position,
      health: bot.health ?? null,
      food: bot.food ?? null,
      is_spawned: isSpawned
    },
    inventory: inventorySummary(bot),
    players: players(bot),
    nearby_entities: nearbyEntities(bot),
    nearby_blocks: nearbyBlocks(bot),
    world: {
      time_of_day: bot.time?.timeOfDay ?? null,
      dimension: bot.game?.dimension ?? 'unknown',
      weather: 'unknown'
    }
  }
}

module.exports = { getState, compactPosition, USEFUL_BLOCKS }
