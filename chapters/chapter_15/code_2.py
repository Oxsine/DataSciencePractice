import argparse
import gymnasium as gym
from typing import Tuple, Any

def setup_arguments():
    """Конфигурация параметров командной строки"""
    arg_parser = argparse.ArgumentParser(
        description='Запуск обучающего агента в различных средах'
    )
    arg_parser.add_argument(
        '--environment',
        dest='env_type',
        required=True,
        choices=['stabilizer', 'climber', 'swinger'],
        help='Выбор среды: stabilizer (стабилизатор), climber (скалолаз), swinger (маятник)'
    )
    arg_parser.add_argument(
        '--episodes',
        type=int,
        default=15,
        help='Количество обучающих эпизодов'
    )
    arg_parser.add_argument(
        '--max-steps',
        type=int,
        default=120,
        help='Максимальное количество шагов в эпизоде'
    )
    return arg_parser

class SimpleRLAgent:
    """Базовый агент обучения с подкреплением"""
    
    def __init__(self, sim_env):
        self.env_instance = sim_env
        self.rewards_log = []
    
    def process_episode(self, max_iterations: int, episode_idx: int) -> Tuple[float, int]:
        """Обработка одного эпизода обучения"""
        print(f"\nЭпизод {episode_idx}:")
        print("-" * 40)
        
        # Инициализация состояния
        curr_state, env_data = self.env_instance.reset()
        episode_return = 0.0
        
        # Последовательность действий
        for step_counter in range(max_iterations):
            # Визуализация
            self.env_instance.render()
            
            # Превью состояния
            state_preview = curr_state[:3] if len(curr_state) > 3 else curr_state
            print(f"Шаг {step_counter:3d}: Состояние {state_preview}")
            
            # Выбор действия
            chosen_action = self.env_instance.action_space.sample()
            print(f"      Действие: {chosen_action}")
            
            # Применение действия
            next_observation, reward_value, done_flag, truncated_flag, extra = self.env_instance.step(
                chosen_action
            )
            
            # Аккумуляция награды
            episode_return += reward_value
            print(f"      Награда: {reward_value:+.4f}, Сумма: {episode_return:.2f}")
            
            # Проверка завершения
            if done_flag or truncated_flag:
                print(f"Эпизод завершен на шаге {step_counter}")
                return episode_return, step_counter
            
            curr_state = next_observation
        
        print(f"Достигнут максимум шагов ({max_iterations})")
        return episode_return, max_iterations

def launch_training():
    """Основная процедура обучения"""
    # Парсинг аргументов
    cmd_args = setup_arguments().parse_args()
    
    # Маппинг имен сред (оставлено как в оригинале)
    environment_mapping = {
        'stabilizer': 'CartPole-v1',
        'climber': 'MountainCar-v0',
        'swinger': 'Pendulum-v1'
    }
    
    # Информация о запуске
    print(f"\n{'='*60}")
    print(f"СТАРТ ОБУЧЕНИЯ АГЕНТА")
    print(f"{'='*60}")
    print(f"Целевая среда: {cmd_args.env_type}")
    print(f"Число эпизодов: {cmd_args.episodes}")
    print(f"Лимит шагов на эпизод: {cmd_args.max_steps}")
    
    try:
        training_environment = gym.make(environment_mapping[cmd_args.env_type], render_mode='human')
        print(f"Режим: с графической визуализацией")
    except Exception as err:
        print(f"Визуализация недоступна: {err}")
        training_environment = gym.make(environment_mapping[cmd_args.env_type], render_mode=None)
        print(f"Режим: без визуализации")
    
    # Создание и запуск агента
    learning_agent = SimpleRLAgent(training_environment)
    
    total_iterations = 0
    episode_rewards = []
    
    # Цикл обучения
    for episode_num in range(1, cmd_args.episodes + 1):
        ep_reward, steps_count = learning_agent.process_episode(cmd_args.max_steps, episode_num)
        episode_rewards.append(ep_reward)
        total_iterations += steps_count
    
    # Закрытие среды
    training_environment.close()
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print(f"{'='*60}")
    print(f"Всего эпизодов: {cmd_args.episodes}")
    print(f"Общее число шагов: {total_iterations}")
    avg_reward = sum(episode_rewards)/len(episode_rewards)
    print(f"Средняя награда за эпизод: {avg_reward:.2f}")
    print(f"Лучшая награда: {max(episode_rewards):.2f}")
    print(f"Худшая награда: {min(episode_rewards):.2f}")
    
    # Анализ по средам
    if cmd_args.env_type == 'stabilizer':
        print(f"\nАНАЛИЗ ДЛЯ СТАБИЛИЗАТОРА:")
        avg_steps = total_iterations/cmd_args.episodes
        print(f"Средняя продолжительность эпизода: {avg_steps:.1f} шагов")
        print("Цель: максимально долго удерживать стержень в вертикальном положении.")
        
    elif cmd_args.env_type == 'climber':
        print(f"\nАНАЛИЗ ДЛЯ СКАЛОЛАЗА:")
        print("Цель: достижение вершины горы с использованием раскачки.")
        positive_rewards = sum(1 for r in episode_rewards if r > 0)
        print(f"Успешных попыток: {positive_rewards} из {cmd_args.episodes}")
        
    elif cmd_args.env_type == 'swinger':
        print(f"\nАНАЛИЗ ДЛЯ МАЯТНИКА:")
        print("Цель: стабилизация маятника в верхнем положении.")
    
    return 0

if __name__ == '__main__':
    exit_status = launch_training()
    exit(exit_status)