import type { CollectionEntry } from 'astro:content';

type Answer = CollectionEntry<'answers'>;

/** topic 徽章的展示颜色（与主题相关联，未命中的回落到灰色） */
export const TOPIC_META: Record<string, { color: string }> = {
  总览: { color: '#3B82F6' },
  'CUDA 基础': { color: '#06B6D4' },
  算子优化: { color: '#F59E0B' },
  推理优化: { color: '#8B5CF6' },
  量化: { color: '#EC4899' },
  训练优化: { color: '#10B981' },
  'C++ 基础': { color: '#6366F1' },
  'C++ 编程': { color: '#6366F1' },
  算法: { color: '#EF4444' },
  // 公司题解批次 (20-104)
  '公司题解-字节跳动': { color: '#0EA5E9' },
  '公司题解-阿里巴巴': { color: '#F97316' },
  '公司题解-腾讯': { color: '#22C55E' },
  '公司题解-百度': { color: '#3B82F6' },
  '公司题解-T2': { color: '#A855F7' },
  '公司题解-T3': { color: '#14B8A6' },
  '公司题解-T4': { color: '#EAB308' },
  '公司题解-T5': { color: '#EC4899' },
};

export function topicColor(topic: string): string {
  return TOPIC_META[topic]?.color ?? '#6B7280';
}

/** 解答篇阅读顺序：README(order 0) 在前，其余按 order 升序 */
export function flattenAnswerOrder(items: Answer[]): Answer[] {
  return [...items].sort((a, b) => a.data.order - b.data.order);
}
