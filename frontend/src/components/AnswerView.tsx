import { Card, Descriptions, Empty, List, Space, Tag } from "antd";

import type { AnswerPayload, SourceItem } from "../types/api";

interface AnswerViewProps {
  answer: AnswerPayload | null;
}

function renderObjectiveValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return `${value.length} 项`;
  }
  return "结构化对象";
}

function buildObjectiveEntries(answer: AnswerPayload) {
  return Object.entries(answer.objective_data)
    .filter(([key]) => key !== "points")
    .slice(0, 10)
    .map(([key, value]) => ({
      key,
      label: key,
      children: renderObjectiveValue(value),
    }));
}

function sourceLabel(source: SourceItem): string {
  if (!source.value) {
    return source.name;
  }
  return source.name;
}

export function AnswerView({ answer }: AnswerViewProps) {
  if (!answer) {
    return (
      <section className="panel answer-placeholder">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="输入问题后，这里会显示结构化答案、证据来源和约束说明。"
        />
      </section>
    );
  }

  const objectiveEntries = buildObjectiveEntries(answer);

  return (
    <div className="answer-stack">
      <section className="panel summary-panel">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">回答摘要</p>
            <h2>{answer.summary}</h2>
          </div>
          <Space wrap>
            <Tag color="blue">{answer.question_type}</Tag>
            <Tag color={answer.route.need_market_data ? "gold" : "default"}>market</Tag>
            <Tag color={answer.route.need_rag ? "green" : "default"}>rag</Tag>
          </Space>
        </div>
        <p className="request-text">问题：{answer.request_message}</p>
      </section>

      <Card className="panel" bordered={false}>
        <p className="section-kicker">客观数据</p>
        <Descriptions bordered size="small" column={1} items={objectiveEntries} />
      </Card>

      <Card className="panel" bordered={false}>
        <p className="section-kicker">分析说明</p>
        <List
          dataSource={answer.analysis}
          renderItem={(item) => (
            <List.Item className="list-item">
              <span>{item}</span>
            </List.Item>
          )}
        />
      </Card>

      <div className="detail-grid">
        <Card className="panel" bordered={false}>
          <p className="section-kicker">引用来源</p>
          <List
            dataSource={answer.sources}
            locale={{ emptyText: "当前回答未附带来源。" }}
            renderItem={(item) => (
              <List.Item className="list-item">
                <div>
                  <strong>{item.type}</strong>
                  <p>
                    {item.value ? (
                      <a className="source-link" href={item.value} target="_blank" rel="noreferrer">
                        {sourceLabel(item)}
                      </a>
                    ) : (
                      sourceLabel(item)
                    )}
                  </p>
                  {item.value ? <p className="source-url">{item.value}</p> : null}
                </div>
              </List.Item>
            )}
          />
        </Card>

        <Card className="panel" bordered={false}>
          <p className="section-kicker">限制说明</p>
          <List
            dataSource={answer.limitations}
            renderItem={(item) => (
              <List.Item className="list-item">
                <span>{item}</span>
              </List.Item>
            )}
          />
        </Card>
      </div>
    </div>
  );
}
