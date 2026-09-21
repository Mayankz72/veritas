import { TopicView } from "@/components/TopicView";

export default async function TopicPage({ params }: PageProps<"/topics/[id]">) {
  const { id } = await params;
  return (
    <main className="mx-auto max-w-4xl px-4 py-10">
      <TopicView topicId={id} />
    </main>
  );
}
