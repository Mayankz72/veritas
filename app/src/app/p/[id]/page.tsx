import { PublicationView } from "@/components/PublicationView";

export default async function PublicationPage({ params }: PageProps<"/p/[id]">) {
  const { id } = await params;
  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <PublicationView publicationId={id} />
    </main>
  );
}
