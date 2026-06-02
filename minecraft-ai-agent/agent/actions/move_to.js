const { GoalNear } = require('mineflayer-pathfinder').goals
const { withTimeout } = require('./utils')

module.exports = async function moveTo (bot, params) {
  const range = Number(params.range ?? 1)
  const goal = new GoalNear(Number(params.x), Number(params.y), Number(params.z), range)
  await withTimeout(bot.pathfinder.goto(goal), 30000, 'move_to')
  return { ok: true, message: 'Arrived near target position' }
}
