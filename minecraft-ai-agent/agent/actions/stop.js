module.exports = async function stop (bot) {
  bot.pathfinder.setGoal(null)
  bot.clearControlStates()
  return { ok: true, message: 'Stopped current movement and actions' }
}
