const { GoalFollow } = require('mineflayer-pathfinder').goals

module.exports = async function followPlayer (bot, params) {
  const player = bot.players[params.player_name]
  if (!player || !player.entity) {
    throw new Error(`Player not found: ${params.player_name}`)
  }
  bot.pathfinder.setGoal(new GoalFollow(player.entity, Number(params.distance ?? 3)), true)
  return { ok: true, message: `Following ${params.player_name}` }
}
