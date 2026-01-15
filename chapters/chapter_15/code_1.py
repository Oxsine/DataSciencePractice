import argparse
import gymnasium

def setup_parser():
    """Настраивает парсер аргументов командной строки"""
    arg_parser = argparse.ArgumentParser(
        description='Запуск симуляционных сред для обучения с подкреплением',
        epilog='Пример: python rl_experiment.py --env_type cartpole'
    )
    arg_parser.add_argument(
        '--environment', 
        dest='chosen_env',
        required=True,
        choices=['balance_pole', 'mountain_climb', 'pendulum'],
        help='Выберите среду для запуска'
    )
    arg_parser.add_argument(
        '--iterations',
        dest='max_steps',
        type=int,
        default=500,
        help='Количество итераций симуляции'
    )
    return arg_parser

def show_env_details(env_obj, env_name):
    """Выводит информацию о среде"""
    print("\n" + "="*50)
    print(f"СРЕДА ЗАПУЩЕНА: {env_name}")
    print("="*50)
    
    print(f"\nХАРАКТЕРИСТИКИ СРЕДЫ:")
    print(f"   Пространство действий: {env_obj.action_space}")
    print(f"   Тип действий: {env_obj.action_space.__class__.__name__}")
    print(f"   Пространство наблюдений: {env_obj.observation_space}")
    print(f"   Границы наблюдений: {getattr(env_obj.observation_space, 'low', 'Н/Д')} "
          f"- {getattr(env_obj.observation_space, 'high', 'Н/Д')}")

def execute_simulation(simulation_env, total_iterations=500):
    """Запускает симуляцию в выбранной среде"""
    print(f"\nЗАПУСК СИМУЛЯЦИИ ({total_iterations} итераций)")
    print("-" * 40)
    
    # Инициализация начального состояния
    current_observation, env_data = simulation_env.reset()
    episode_number = 1
    cumulative_reward = 0
    actions_count = 0
    
    for step_index in range(total_iterations):
        # Выбор случайного действия
        action_taken = simulation_env.action_space.sample()
        
        # Выполнение действия в среде
        next_observation, reward_gained, is_done, is_truncated, extra_data = simulation_env.step(action_taken)
        
        # Накопление награды
        cumulative_reward += reward_gained
        actions_count += 1
        
        # Периодический вывод информации
        if step_index % 75 == 0:
            print(f"   Шаг {step_index:3d} | "
                  f"Действие: {action_taken} | "
                  f"Награда: {reward_gained:+.3f} | "
                  f"Сумма: {cumulative_reward:.1f}")
        
        # Проверка завершения эпизода
        if is_done or is_truncated:
            print(f"\n Эпизод {episode_number} завершен!")
            print(f"   Шагов: {actions_count} | "
                  f"Финальная награда: {cumulative_reward:.2f}")
            
            # Сброс для нового эпизода
            current_observation, env_data = simulation_env.reset()
            episode_number += 1
            cumulative_reward = 0
            actions_count = 0
        else:
            current_observation = next_observation
    
    print(f"\nСИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print(f"   Всего эпизодов: {episode_number-1}")
    return episode_number - 1

def run_program():
    """Основная функция программы"""
    # Парсинг аргументов
    parser_instance = setup_parser()
    parsed_args = parser_instance.parse_args()
    
    # Маппинг имен сред
    env_name_mapping = {
        'balance_pole': 'CartPole-v1',
        'mountain_climb': 'MountainCar-v0',
        'pendulum': 'Pendulum-v1'
    }
    
    try:
        print(f"\nПодготовка среды: {parsed_args.chosen_env}")
        
        # Создаем среду
        environment = gymnasium.make(
            env_name_mapping[parsed_args.chosen_env],
            render_mode='human'
        )
        
        # Выводим информацию о среде
        show_env_details(environment, env_name_mapping[parsed_args.chosen_env])
        
        # Запускаем симуляцию
        episodes_completed = execute_simulation(environment, parsed_args.max_steps)
        
        # Закрываем среду
        environment.close()
        
        print(f"\nИТОГИ ЭКСПЕРИМЕНТА:")
        print(f"  Среда: {parsed_args.chosen_env.replace('_', ' ').title()}")
        print(f"  Итераций выполнено: {parsed_args.max_steps}")
        print(f"  Эпизодов завершено: {episodes_completed}")
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    result_code = run_program()
    exit(result_code)